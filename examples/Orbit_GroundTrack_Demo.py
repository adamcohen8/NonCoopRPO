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
    j2_plugin,
)
from sim.utils.ground_track import ground_track_from_eci_history
from sim.utils.plotting import plot_ground_track


def _circular_state(radius_km: float, inc_deg: float, raan_deg: float) -> np.ndarray:
    inc = np.deg2rad(float(inc_deg))
    raan = np.deg2rad(float(raan_deg))
    speed = np.sqrt(EARTH_MU_KM3_S2 / radius_km)
    r_pqw = np.array([radius_km, 0.0, 0.0], dtype=float)
    v_pqw = np.array([0.0, speed, 0.0], dtype=float)

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    rot = np.array(
        [
            [cO, -sO * ci, sO * si],
            [sO, cO * ci, -cO * si],
            [0.0, si, ci],
        ],
        dtype=float,
    )
    r = rot @ r_pqw
    v = rot @ v_pqw
    return np.hstack((r, v))


def run_demo(
    plot_mode: str = "interactive",
    dt_s: float = 20.0,
    orbits: float = 3.0,
    altitude_km: float = 550.0,
    inc_deg: float = 53.0,
    use_j2: bool = True,
) -> dict[str, str]:
    radius = 6378.137 + float(altitude_km)
    x0 = _circular_state(radius_km=radius, inc_deg=inc_deg, raan_deg=25.0)
    period_s = 2.0 * np.pi * np.sqrt((radius**3) / EARTH_MU_KM3_S2)
    duration_s = float(orbits) * period_s
    steps = int(np.ceil(duration_s / float(dt_s)))
    t = np.arange(steps + 1, dtype=float) * float(dt_s)

    ctx = OrbitContext(mu_km3_s2=EARTH_MU_KM3_S2, mass_kg=200.0, area_m2=1.0, cd=2.2, cr=1.2)
    plugins = [j2_plugin] if bool(use_j2) else []
    prop = OrbitPropagator(integrator="rk4", plugins=plugins)

    x = np.zeros((steps + 1, 6), dtype=float)
    x[0, :] = x0
    for k in range(steps):
        x[k + 1, :] = prop.propagate(
            x_eci=x[k, :],
            dt_s=float(dt_s),
            t_s=float(t[k]),
            command_accel_eci_km_s2=np.zeros(3),
            env={},
            ctx=ctx,
        )

    jd0 = datetime_to_julian_date(datetime(2026, 3, 11, 0, 0, 0, tzinfo=timezone.utc))
    lat, lon, alt = ground_track_from_eci_history(x[:, :3], t_s=t, jd_utc_start=jd0)

    outdir = REPO_ROOT / "outputs" / "orbit_groundtrack_demo"
    gt_path = outdir / "ground_track.png"
    alt_path = outdir / "ground_track_altitude.png"
    if plot_mode in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)

    plot_ground_track(
        lon_deg=lon,
        lat_deg=lat,
        mode=plot_mode,
        out_path=str(gt_path),
        title=f"Ground Track ({'J2' if use_j2 else 'Two-Body'})",
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t / 3600.0, alt)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Altitude vs Time")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if plot_mode in ("save", "both"):
        fig.savefig(alt_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    return {
        "plot_mode": plot_mode,
        "duration_hours": f"{duration_s / 3600.0:.3f}",
        "lat_min_deg": f"{float(np.min(lat)):.6f}",
        "lat_max_deg": f"{float(np.max(lat)):.6f}",
        "alt_min_km": f"{float(np.min(alt)):.6f}",
        "alt_max_km": f"{float(np.max(alt)):.6f}",
        "ground_track_plot": str(gt_path) if plot_mode in ("save", "both") else "",
        "altitude_plot": str(alt_path) if plot_mode in ("save", "both") else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orbit ground-track demo from propagated ECI orbit history.")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--dt", type=float, default=20.0)
    parser.add_argument("--orbits", type=float, default=3.0)
    parser.add_argument("--altitude-km", type=float, default=550.0)
    parser.add_argument("--inc-deg", type=float, default=53.0)
    parser.add_argument("--no-j2", action="store_true", help="Use pure two-body propagation.")
    args = parser.parse_args()

    out = run_demo(
        plot_mode=args.plot_mode,
        dt_s=float(args.dt),
        orbits=float(args.orbits),
        altitude_km=float(args.altitude_km),
        inc_deg=float(args.inc_deg),
        use_j2=not bool(args.no_j2),
    )
    print("Ground-track demo outputs:")
    for k, v in out.items():
        if v:
            print(f"  {k}: {v}")
