import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.config import get_simulation_profile, profile_choices, resolve_dt_s
from sim.dynamics.orbit.propagator import (
    OrbitPropagator,
    drag_plugin,
    j2_plugin,
    j3_plugin,
    j4_plugin,
    srp_plugin,
    third_body_moon_plugin,
    third_body_sun_plugin,
)
from sim.dynamics.orbit.accelerations import OrbitContext


def _circular_orbit_state_eci(radius_km: float) -> np.ndarray:
    speed_km_s = np.sqrt(EARTH_MU_KM3_S2 / radius_km)
    return np.array([radius_km, 0.0, 0.0, 0.0, speed_km_s, 0.0], dtype=float)


def _propagate(
    x0: np.ndarray,
    duration_s: float,
    dt_s: float,
    ctx: OrbitContext,
    env: dict,
    plugins: list,
    integrator: str,
    adaptive_atol: float,
    adaptive_rtol: float,
) -> tuple[np.ndarray, np.ndarray]:
    steps = int(np.ceil(duration_s / dt_s))
    t = np.arange(steps + 1, dtype=float) * dt_s
    x = np.zeros((steps + 1, 6), dtype=float)
    x[0, :] = x0

    propagator = OrbitPropagator(
        integrator=integrator,
        plugins=list(plugins),
        adaptive_atol=adaptive_atol,
        adaptive_rtol=adaptive_rtol,
    )
    zero_cmd = np.zeros(3, dtype=float)
    for k in range(steps):
        x[k + 1, :] = propagator.propagate(
            x_eci=x[k, :],
            dt_s=dt_s,
            t_s=t[k],
            command_accel_eci_km_s2=zero_cmd,
            env=env,
            ctx=ctx,
        )
    return t, x


def run_demo(
    plot_mode: str = "interactive",
    dt_s: float | None = None,
    altitude_km: float = 500.0,
    profile: str = "ops",
) -> dict[str, str]:
    p = get_simulation_profile(profile)
    dt_used_s = resolve_dt_s(profile, dt_s)
    radius_km = EARTH_RADIUS_KM + altitude_km
    x0 = _circular_orbit_state_eci(radius_km)
    period_s = 2.0 * np.pi * np.sqrt((radius_km**3) / EARTH_MU_KM3_S2)

    ctx = OrbitContext(
        mu_km3_s2=EARTH_MU_KM3_S2,
        mass_kg=120.0,
        area_m2=1.5,
        cd=2.2,
        cr=1.3,
    )
    env = {
        "density_kg_m3": 1.0e-12,
        "sun_dir_eci": np.array([1.0, 0.0, 0.0], dtype=float),
        "moon_pos_eci_km": np.array([384400.0, 0.0, 0.0], dtype=float),
        "sun_pos_eci_km": np.array([149597870.7, 0.0, 0.0], dtype=float),
    }

    t, x_base = _propagate(
        x0,
        period_s,
        dt_used_s,
        ctx,
        env,
        plugins=[],
        integrator=p.orbit_integrator,
        adaptive_atol=p.orbit_adaptive_atol,
        adaptive_rtol=p.orbit_adaptive_rtol,
    )
    perturbation_cases = {
        "J2": [j2_plugin],
        "J3": [j3_plugin],
        "J4": [j4_plugin],
        "Drag": [drag_plugin],
        "SRP": [srp_plugin],
        "Moon 3rd Body": [third_body_moon_plugin],
        "Sun 3rd Body": [third_body_sun_plugin],
    }

    case_results: dict[str, np.ndarray] = {}
    for name, plugins in perturbation_cases.items():
        _, x_case = _propagate(
            x0,
            period_s,
            dt_used_s,
            ctx,
            env,
            plugins=plugins,
            integrator=p.orbit_integrator,
            adaptive_atol=p.orbit_adaptive_atol,
            adaptive_rtol=p.orbit_adaptive_rtol,
        )
        case_results[name] = x_case

    err_pos_norm: dict[str, np.ndarray] = {}
    err_vel_norm: dict[str, np.ndarray] = {}
    for name, x_case in case_results.items():
        dx = x_case - x_base
        err_pos_norm[name] = np.linalg.norm(dx[:, :3], axis=1)
        err_vel_norm[name] = np.linalg.norm(dx[:, 3:], axis=1)

    outdir = REPO_ROOT / "outputs" / "orbit_perturbation_error_demo"
    if plot_mode in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)

    fig_base, axes_base = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes_base[0].plot(t, x_base[:, 0], label="x")
    axes_base[1].plot(t, x_base[:, 1], label="y")
    axes_base[2].plot(t, x_base[:, 2], label="z")
    axes_base[0].set_title("Baseline One-Orbit Propagation (Two-Body, No Perturbations)")
    axes_base[0].set_ylabel("x (km)")
    axes_base[1].set_ylabel("y (km)")
    axes_base[2].set_ylabel("z (km)")
    axes_base[2].set_xlabel("Time (s)")
    for ax in axes_base:
        ax.grid(True, alpha=0.3)
    fig_base.tight_layout()
    if plot_mode in ("save", "both"):
        fig_base.savefig(outdir / "baseline_two_body_orbit.png", dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_base)

    fig_sum, axes_sum = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for name in perturbation_cases:
        axes_sum[0].plot(t, err_pos_norm[name], label=name)
        axes_sum[1].plot(t, err_vel_norm[name], label=name)
    axes_sum[0].set_title("State Error vs Two-Body Baseline (One Orbit)")
    axes_sum[0].set_ylabel("Position Error Norm (km)")
    axes_sum[1].set_ylabel("Velocity Error Norm (km/s)")
    axes_sum[1].set_xlabel("Time (s)")
    axes_sum[0].legend(loc="best")
    for ax in axes_sum:
        ax.grid(True, alpha=0.3)
    fig_sum.tight_layout()
    if plot_mode in ("save", "both"):
        fig_sum.savefig(outdir / "perturbation_error_summary.png", dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_sum)

    case_names = list(perturbation_cases.keys())
    fig_ind, axes_ind = plt.subplots(len(case_names), 2, figsize=(13, 2.5 * len(case_names)), sharex=True)
    if len(case_names) == 1:
        axes_ind = np.array([axes_ind])
    for i, name in enumerate(case_names):
        axes_ind[i, 0].plot(t, err_pos_norm[name], color="tab:blue")
        axes_ind[i, 1].plot(t, err_vel_norm[name], color="tab:orange")
        axes_ind[i, 0].set_ylabel(f"{name}\n|dr| (km)")
        axes_ind[i, 1].set_ylabel("|dv| (km/s)")
        axes_ind[i, 0].grid(True, alpha=0.3)
        axes_ind[i, 1].grid(True, alpha=0.3)
    axes_ind[0, 0].set_title("Position Error Norm")
    axes_ind[0, 1].set_title("Velocity Error Norm")
    axes_ind[-1, 0].set_xlabel("Time (s)")
    axes_ind[-1, 1].set_xlabel("Time (s)")
    fig_ind.suptitle("Single-Perturbation Error Growth vs Two-Body Baseline", y=0.998)
    fig_ind.tight_layout()
    if plot_mode in ("save", "both"):
        fig_ind.savefig(outdir / "perturbation_error_individual.png", dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_ind)

    outputs = {
        "plot_mode": plot_mode,
        "baseline_plot": str(outdir / "baseline_two_body_orbit.png") if plot_mode in ("save", "both") else "",
        "summary_plot": str(outdir / "perturbation_error_summary.png") if plot_mode in ("save", "both") else "",
        "individual_plot": str(outdir / "perturbation_error_individual.png") if plot_mode in ("save", "both") else "",
    }
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run one-orbit two-body baseline and one-at-a-time perturbation error comparisons."
    )
    parser.add_argument(
        "--plot-mode",
        choices=["interactive", "save", "both"],
        default="interactive",
        help="Plot behavior; interactive is default.",
    )
    parser.add_argument("--dt", type=float, default=None, help="Integrator step size in seconds (overrides profile).")
    parser.add_argument("--altitude-km", type=float, default=500.0, help="Initial circular orbit altitude in km.")
    parser.add_argument(
        "--profile",
        choices=list(profile_choices()),
        default="ops",
        help="Fidelity profile: fast, ops, or high_fidelity.",
    )
    args = parser.parse_args()

    result = run_demo(
        plot_mode=args.plot_mode,
        dt_s=None if args.dt is None else float(args.dt),
        altitude_km=float(args.altitude_km),
        profile=args.profile,
    )
    print("Generated outputs:")
    for k, v in result.items():
        if v:
            print(f"  {k}: {v}")
