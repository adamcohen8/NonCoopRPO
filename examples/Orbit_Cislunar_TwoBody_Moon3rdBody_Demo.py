from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.dynamics.orbit import (
    EARTH_MU_KM3_S2,
    OrbitContext,
    OrbitPropagator,
    datetime_to_julian_date,
    resolve_time_dependent_env,
    third_body_moon_plugin,
)


def _initial_cislunar_state_eci_km_s(perigee_km: float, apogee_km: float) -> np.ndarray:
    a = 0.5 * (perigee_km + apogee_km)
    v_perigee = np.sqrt(EARTH_MU_KM3_S2 * (2.0 / perigee_km - 1.0 / a))
    return np.array([perigee_km, 0.0, 0.0, 0.0, v_perigee, 0.0], dtype=float)


def run_demo(
    plot_mode: str = "interactive",
    dt_s: float = 120.0,
    duration_days: float = 14.0,
    perigee_km: float = 6678.137,
    apogee_km: float = 380000.0,
) -> dict[str, str]:
    x0 = _initial_cislunar_state_eci_km_s(perigee_km=perigee_km, apogee_km=apogee_km)
    duration_s = float(duration_days) * 86400.0
    steps = int(np.ceil(duration_s / float(dt_s)))
    t = np.arange(steps + 1, dtype=float) * float(dt_s)

    x = np.zeros((steps + 1, 6), dtype=float)
    x[0, :] = x0

    jd0 = datetime_to_julian_date(datetime(2026, 3, 11, 0, 0, 0, tzinfo=timezone.utc))
    env_base = {
        "jd_utc_start": jd0,
        "ephemeris_mode": "analytic_enhanced",
    }

    ctx = OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=500.0, area_m2=1.0, cd=2.2, cr=1.2)
    prop = OrbitPropagator(integrator="adaptive", plugins=[third_body_moon_plugin], adaptive_atol=1e-9, adaptive_rtol=1e-8)

    for k in range(steps):
        tk = t[k]
        env_k = resolve_time_dependent_env(env_base, float(tk))
        x[k + 1, :] = prop.propagate(
            x_eci=x[k, :],
            dt_s=float(dt_s),
            t_s=float(tk),
            command_accel_eci_km_s2=np.zeros(3),
            env=env_k,
            ctx=ctx,
        )

    moon_hist = np.zeros((steps + 1, 3), dtype=float)
    for k in range(steps + 1):
        env_k = resolve_time_dependent_env(env_base, float(t[k]))
        moon_hist[k, :] = np.array(env_k["moon_pos_eci_km"], dtype=float)

    r_norm = np.linalg.norm(x[:, :3], axis=1)
    moon_dist = np.linalg.norm(x[:, :3] - moon_hist, axis=1)

    outdir = REPO_ROOT / "outputs" / "orbit_cislunar_two_body_moon3rd"
    if plot_mode in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x[:, 0], x[:, 1], x[:, 2], label="Spacecraft (2-body + Moon 3rd)")
    ax.plot(moon_hist[:, 0], moon_hist[:, 1], moon_hist[:, 2], "--", alpha=0.7, label="Moon (analytic)")
    ax.scatter([x[0, 0]], [x[0, 1]], [x[0, 2]], c="tab:green", s=30, label="Start")
    ax.scatter([0.0], [0.0], [0.0], c="tab:blue", s=50, label="Earth")
    ax.set_title("Cislunar Orbit Propagation (Earth 2-Body + Moon 3rd Body)")
    ax.set_xlabel("X ECI (km)")
    ax.set_ylabel("Y ECI (km)")
    ax.set_zlabel("Z ECI (km)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p_orbit = outdir / "cislunar_eci_3d.png"
    if plot_mode in ("save", "both"):
        fig.savefig(p_orbit, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    fig2, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t / 86400.0, r_norm)
    axes[0].set_ylabel("|r_eci| (km)")
    axes[0].set_title("Cislunar Range Metrics")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t / 86400.0, moon_dist)
    axes[1].set_ylabel("|r_sc - r_moon| (km)")
    axes[1].set_xlabel("Time (days)")
    axes[1].grid(True, alpha=0.3)
    fig2.tight_layout()
    p_range = outdir / "cislunar_range_metrics.png"
    if plot_mode in ("save", "both"):
        fig2.savefig(p_range, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig2)

    return {
        "plot_mode": plot_mode,
        "duration_days": f"{duration_days:.3f}",
        "dt_s": f"{dt_s:.3f}",
        "r_eci_min_km": f"{float(np.min(r_norm)):.3f}",
        "r_eci_max_km": f"{float(np.max(r_norm)):.3f}",
        "moon_range_min_km": f"{float(np.min(moon_dist)):.3f}",
        "moon_range_max_km": f"{float(np.max(moon_dist)):.3f}",
        "orbit_plot": str(p_orbit) if plot_mode in ("save", "both") else "",
        "range_plot": str(p_range) if plot_mode in ("save", "both") else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cislunar orbit demo using Earth two-body + Moon third-body perturbation.")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--dt", type=float, default=120.0, help="Integrator step size (s).")
    parser.add_argument("--duration-days", type=float, default=14.0, help="Total simulation duration (days).")
    parser.add_argument("--perigee-km", type=float, default=6678.137, help="Initial perigee radius from Earth center (km).")
    parser.add_argument("--apogee-km", type=float, default=380000.0, help="Initial apogee radius from Earth center (km).")
    args = parser.parse_args()

    out = run_demo(
        plot_mode=args.plot_mode,
        dt_s=float(args.dt),
        duration_days=float(args.duration_days),
        perigee_km=float(args.perigee_km),
        apogee_km=float(args.apogee_km),
    )
    print("Cislunar demo outputs:")
    for k, v in out.items():
        if v:
            print(f"  {k}: {v}")
