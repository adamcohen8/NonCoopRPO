from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
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
    j2_plugin,
    j3_plugin,
    j4_plugin,
    resolve_time_dependent_env,
    srp_plugin,
    third_body_moon_plugin,
    third_body_sun_plugin,
)


def _circular_orbit_state_eci(radius_km: float) -> np.ndarray:
    speed_km_s = np.sqrt(EARTH_MU_KM3_S2 / radius_km)
    return np.array([radius_km, 0.0, 0.0, 0.0, speed_km_s, 0.0], dtype=float)


def _require_nrlmsise00_backend() -> None:
    if importlib.util.find_spec("nrlmsise00") is None:
        raise RuntimeError(
            "This demo requires the optional `nrlmsise00` package for drag with the NRLMSISE-00 atmosphere model."
        )


def _propagate(
    *,
    x0: np.ndarray,
    duration_s: float,
    dt_s: float,
    integrator: str,
    adaptive_atol: float,
    adaptive_rtol: float,
    env_base: dict,
    ctx: OrbitContext,
) -> tuple[np.ndarray, np.ndarray]:
    steps = int(np.ceil(duration_s / dt_s))
    t = np.arange(steps + 1, dtype=float) * dt_s
    x = np.zeros((steps + 1, 6), dtype=float)
    x[0, :] = x0

    prop = OrbitPropagator(
        integrator=integrator,
        plugins=[
            j2_plugin,
            j3_plugin,
            j4_plugin,
            drag_plugin,
            third_body_sun_plugin,
            third_body_moon_plugin,
            srp_plugin,
        ],
        adaptive_atol=adaptive_atol,
        adaptive_rtol=adaptive_rtol,
    )
    zero_cmd = np.zeros(3, dtype=float)
    for k in range(steps):
        tk = float(t[k])
        env_k = resolve_time_dependent_env(env_base, tk)
        x[k + 1, :] = prop.propagate(
            x_eci=x[k, :],
            dt_s=float(dt_s),
            t_s=tk,
            command_accel_eci_km_s2=zero_cmd,
            env=env_k,
            ctx=ctx,
        )
    return t, x


def run_demo(
    *,
    plot_mode: str = "interactive",
    altitude_km: float = 500.0,
    dt_s: float = 1.0,
    rkf78_atol: float = 1.0e-12,
    rkf78_rtol: float = 1.0e-10,
) -> dict[str, str]:
    _require_nrlmsise00_backend()
    import matplotlib.pyplot as plt

    radius_km = EARTH_RADIUS_KM + float(altitude_km)
    x0 = _circular_orbit_state_eci(radius_km)
    period_s = 2.0 * np.pi * np.sqrt((radius_km**3) / EARTH_MU_KM3_S2)
    start_epoch = datetime(2026, 3, 11, 0, 0, 0, tzinfo=timezone.utc)

    ctx = OrbitContext(
        mu_km3_s2=EARTH_MU_KM3_S2,
        mass_kg=120.0,
        area_m2=2.0,
        cd=2.2,
        cr=1.3,
    )
    env_base = {
        "jd_utc_start": datetime_to_julian_date(start_epoch),
        "ephemeris_mode": "analytic_enhanced",
        "atmosphere_model": "nrlmsise00",
        "geodetic_model": "wgs84",
        "f107a": 150.0,
        "f107": 150.0,
        "ap": 4.0,
        "srp_shadow_model": "conical",
    }

    t_rk4, x_rk4 = _propagate(
        x0=x0,
        duration_s=5*period_s,
        dt_s=dt_s,
        integrator="rk4",
        adaptive_atol=rkf78_atol,
        adaptive_rtol=rkf78_rtol,
        env_base=env_base,
        ctx=ctx,
    )
    t_rkf78, x_rkf78 = _propagate(
        x0=x0,
        duration_s=5*period_s,
        dt_s=dt_s,
        integrator="rkf78",
        adaptive_atol=rkf78_atol,
        adaptive_rtol=rkf78_rtol,
        env_base=env_base,
        ctx=ctx,
    )

    if not np.array_equal(t_rk4, t_rkf78):
        raise RuntimeError("RK4 and RKF78 runs produced different time grids; this demo expects a shared outer step.")

    dx = x_rk4 - x_rkf78
    pos_err_km = np.linalg.norm(dx[:, :3], axis=1)
    vel_err_km_s = np.linalg.norm(dx[:, 3:], axis=1)

    outdir = REPO_ROOT / "outputs" / "orbit_rk4_vs_rkf78_high_fidelity_demo"
    if plot_mode in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t_rk4 / 60.0, pos_err_km, label="|r_RK4 - r_RKF78|")
    axes[0].set_ylabel("Position Error (km)")
    axes[0].set_title("One-Orbit Integrator Error: RK4 vs RKF78")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t_rk4 / 60.0, vel_err_km_s, color="tab:orange", label="|v_RK4 - v_RKF78|")
    axes[1].set_ylabel("Velocity Error (km/s)")
    axes[1].set_xlabel("Time (min)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()

    error_plot = outdir / "rk4_vs_rkf78_error_over_time.png"
    if plot_mode in ("save", "both"):
        fig.savefig(error_plot, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    fig3d = plt.figure(figsize=(10, 8))
    ax = fig3d.add_subplot(111, projection="3d")
    ax.plot(x_rk4[:, 0], x_rk4[:, 1], x_rk4[:, 2], "--", linewidth=1.2, label="RK4")
    ax.plot(x_rkf78[:, 0], x_rkf78[:, 1], x_rkf78[:, 2], linewidth=1.4, label="RKF78")
    ax.set_title("One-Orbit Trajectory Overlay with J2/J3/J4 + Drag(NRL) + Sun/Moon + SRP")
    ax.set_xlabel("X ECI (km)")
    ax.set_ylabel("Y ECI (km)")
    ax.set_zlabel("Z ECI (km)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig3d.tight_layout()

    overlay_plot = outdir / "rk4_vs_rkf78_trajectory_overlay.png"
    if plot_mode in ("save", "both"):
        fig3d.savefig(overlay_plot, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig3d)

    return {
        "plot_mode": plot_mode,
        "dt_s": f"{dt_s:.3f}",
        "period_s": f"{period_s:.3f}",
        "max_position_error_km": f"{float(np.max(pos_err_km)):.9f}",
        "max_velocity_error_km_s": f"{float(np.max(vel_err_km_s)):.12f}",
        "error_plot": str(error_plot) if plot_mode in ("save", "both") else "",
        "overlay_plot": str(overlay_plot) if plot_mode in ("save", "both") else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare RK4 and RKF78 over one orbit with J2/J3/J4, NRL drag, Sun/Moon third-body, and SRP."
    )
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--altitude-km", type=float, default=500.0, help="Initial circular orbit altitude.")
    parser.add_argument("--dt", type=float, default=1.0, help="Outer simulator/integrator step size in seconds.")
    parser.add_argument("--rkf78-atol", type=float, default=1.0e-12, help="RKF78 absolute tolerance.")
    parser.add_argument("--rkf78-rtol", type=float, default=1.0e-10, help="RKF78 relative tolerance.")
    args = parser.parse_args()

    out = run_demo(
        plot_mode=args.plot_mode,
        altitude_km=float(args.altitude_km),
        dt_s=float(args.dt),
        rkf78_atol=float(args.rkf78_atol),
        rkf78_rtol=float(args.rkf78_rtol),
    )
    print("RK4 vs RKF78 high-fidelity demo outputs:")
    for k, v in out.items():
        if v:
            print(f"  {k}: {v}")
