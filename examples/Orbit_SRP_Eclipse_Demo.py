from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.presets import BASIC_SATELLITE, build_sim_object_from_presets
from sim.core.kernel import SimulationKernel
from sim.core.models import SimConfig
from sim.dynamics.orbit import (
    EARTH_MU_KM3_S2,
    OrbitContext,
    OrbitPropagator,
    accel_srp,
    datetime_to_julian_date,
    resolve_time_dependent_env,
    srp_plugin,
)


def _one_run(
    shadow_model: str,
    dt_s: float,
    duration_s: float,
    initial_jd_utc: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sat = build_sim_object_from_presets(
        object_id=f"sat_srp_{shadow_model}",
        dt_s=dt_s,
        satellite=BASIC_SATELLITE,
        enable_disturbances=False,
        orbit_substep_s=dt_s,
        attitude_substep_s=0.01,
    )
    sat.dynamics = replace(
        sat.dynamics,
        orbit_propagator=OrbitPropagator(integrator="rk4", plugins=[srp_plugin]),
        area_m2=2.0,
        cr=1.3,
    )
    steps = int(np.ceil(duration_s / dt_s))
    kernel = SimulationKernel(
        config=SimConfig(dt_s=dt_s, steps=steps, initial_jd_utc=initial_jd_utc, controller_budget_ms=1.0),
        objects=[sat],
        env={"ephemeris_mode": "analytic_simple", "srp_shadow_model": shadow_model},
    )
    log = kernel.run()
    truth = log.truth_by_object[sat.cfg.object_id]
    t = log.t_s
    shadow = log.srp_shadow_by_object[sat.cfg.object_id]

    a_srp = np.zeros_like(t)
    for k, tk in enumerate(t):
        env_k = resolve_time_dependent_env({"jd_utc_start": initial_jd_utc, "ephemeris_mode": "analytic_simple", "srp_shadow_model": shadow_model}, float(tk))
        r = truth[k, :3]
        m = float(truth[k, 13])
        a = accel_srp(r, m, sat.dynamics.area_m2, sat.dynamics.cr, float(tk), env_k)
        a_srp[k] = np.linalg.norm(a)
    return t, truth, shadow, a_srp


def run_demo(plot_mode: str = "interactive", dt_s: float = 2.0, altitude_km: float = 500.0) -> dict[str, str]:
    radius_km = 6378.137 + float(altitude_km)
    period_s = 2.0 * np.pi * np.sqrt((radius_km**3) / EARTH_MU_KM3_S2)
    initial_jd_utc = datetime_to_julian_date(datetime(2026, 3, 11, 0, 0, 0, tzinfo=timezone.utc))

    t_on, truth_on, shadow_on, a_on = _one_run("conical", dt_s=dt_s, duration_s=period_s, initial_jd_utc=initial_jd_utc)
    t_off, truth_off, shadow_off, a_off = _one_run("none", dt_s=dt_s, duration_s=period_s, initial_jd_utc=initial_jd_utc)

    outdir = REPO_ROOT / "outputs" / "orbit_srp_eclipse_demo"
    if plot_mode in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(t_off, a_off * 1e6, label="SRP accel (no eclipse)")
    axes[0].plot(t_on, a_on * 1e6, label="SRP accel (with eclipse)")
    axes[0].set_ylabel("|a_srp| (mm/s^2)")
    axes[0].set_title("SRP Acceleration: Eclipse Disabled vs Enabled")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(t_on, shadow_on, label="shadow factor (with eclipse)")
    axes[1].plot(t_off, shadow_off, "--", label="shadow factor (disabled)")
    axes[1].set_ylabel("Shadow factor")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")
    fig.tight_layout()

    p1 = outdir / "srp_accel_and_shadow.png"
    if plot_mode in ("save", "both"):
        fig.savefig(p1, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    fig3d = plt.figure(figsize=(9, 8))
    ax = fig3d.add_subplot(111, projection="3d")
    ax.plot(truth_off[:, 0], truth_off[:, 1], truth_off[:, 2], "--", label="No eclipse")
    ax.plot(truth_on[:, 0], truth_on[:, 1], truth_on[:, 2], label="With eclipse")
    ax.set_title("One-Orbit ECI Trajectory: SRP Eclipse Effect")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig3d.tight_layout()
    p2 = outdir / "eci_orbit_overlay.png"
    if plot_mode in ("save", "both"):
        fig3d.savefig(p2, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig3d)

    return {
        "plot_mode": plot_mode,
        "accel_shadow_plot": str(p1) if plot_mode in ("save", "both") else "",
        "eci_overlay_plot": str(p2) if plot_mode in ("save", "both") else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo SRP eclipse modeling impact (umbra+penumbra) over one orbit.")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--dt", type=float, default=2.0)
    parser.add_argument("--altitude-km", type=float, default=500.0)
    args = parser.parse_args()

    out = run_demo(plot_mode=args.plot_mode, dt_s=float(args.dt), altitude_km=float(args.altitude_km))
    print("SRP eclipse demo outputs:")
    for k, v in out.items():
        if v:
            print(f"  {k}: {v}")
