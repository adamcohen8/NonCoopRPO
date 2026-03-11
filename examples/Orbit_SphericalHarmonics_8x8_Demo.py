from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.dynamics.orbit.accelerations import OrbitContext
from sim.config import get_simulation_profile, profile_choices, resolve_dt_s
from sim.dynamics.orbit.environment import EARTH_MU_KM3_S2, EARTH_RADIUS_KM
from sim.dynamics.orbit.propagator import OrbitPropagator, spherical_harmonics_plugin
from sim.dynamics.orbit.spherical_harmonics import load_real_earth_gravity_terms


def _circular_orbit_state_eci(radius_km: float, inclination_deg: float) -> np.ndarray:
    speed_km_s = np.sqrt(EARTH_MU_KM3_S2 / radius_km)
    inc = np.deg2rad(float(inclination_deg))
    # Start at ascending node with argument of latitude = 0.
    return np.array([radius_km, 0.0, 0.0, 0.0, speed_km_s * np.cos(inc), speed_km_s * np.sin(inc)], dtype=float)


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
    inclination_deg: float = 45.0,
    orbits: float = 1.0,
    gravity_model: str = "EGM96",
    coeff_path: str = "",
    allow_download: bool = True,
    fd_step_km: float = 1.0e-3,
    profile: str = "ops",
) -> dict[str, str]:
    p = get_simulation_profile(profile)
    dt_used_s = resolve_dt_s(profile, dt_s)
    radius_km = EARTH_RADIUS_KM + float(altitude_km)
    x0 = _circular_orbit_state_eci(radius_km, inclination_deg=float(inclination_deg))
    period_s = 2.0 * np.pi * np.sqrt((radius_km**3) / EARTH_MU_KM3_S2)
    duration_s = 5 * float(orbits) * period_s

    ctx = OrbitContext(
        mu_km3_s2=EARTH_MU_KM3_S2,
        mass_kg=120.0,
        area_m2=1.5,
        cd=2.2,
        cr=1.3,
    )

    t, x_two_body = _propagate(
        x0=x0,
        duration_s=duration_s,
        dt_s=float(dt_used_s),
        ctx=ctx,
        env={},
        plugins=[],
        integrator=p.orbit_integrator,
        adaptive_atol=p.orbit_adaptive_atol,
        adaptive_rtol=p.orbit_adaptive_rtol,
    )

    terms = load_real_earth_gravity_terms(
        max_degree=8,
        max_order=8,
        model=str(gravity_model),
        coeff_path=None if not coeff_path else str(coeff_path),
        allow_download=bool(allow_download),
    )
    env_sh = {
        "spherical_harmonics_terms": terms,
        "spherical_harmonics_fd_step_km": float(fd_step_km),
    }
    _, x_sh = _propagate(
        x0=x0,
        duration_s=duration_s,
        dt_s=float(dt_used_s),
        ctx=ctx,
        env=env_sh,
        plugins=[spherical_harmonics_plugin],
        integrator=p.orbit_integrator,
        adaptive_atol=p.orbit_adaptive_atol,
        adaptive_rtol=p.orbit_adaptive_rtol,
    )

    err = x_sh - x_two_body
    pos_err_norm_m = np.linalg.norm(err[:, :3], axis=1) * 1e3
    vel_err_norm_mm_s = np.linalg.norm(err[:, 3:], axis=1) * 1e6

    outdir = REPO_ROOT / "outputs" / "orbit_spherical_harmonics_8x8_demo"
    if plot_mode in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)

    fig_state, axes_state = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    pos_labels = ["x (km)", "y (km)", "z (km)"]
    vel_labels = ["vx (km/s)", "vy (km/s)", "vz (km/s)"]
    for i in range(3):
        axes_state[i, 0].plot(t, x_two_body[:, i], "--", linewidth=1.0, label="Two-body")
        axes_state[i, 0].plot(t, x_sh[:, i], linewidth=1.4, label="8x8 SH")
        axes_state[i, 0].set_ylabel(pos_labels[i])
        axes_state[i, 0].grid(True, alpha=0.3)
        axes_state[i, 1].plot(t, x_two_body[:, i + 3], "--", linewidth=1.0, label="Two-body")
        axes_state[i, 1].plot(t, x_sh[:, i + 3], linewidth=1.4, label="8x8 SH")
        axes_state[i, 1].set_ylabel(vel_labels[i])
        axes_state[i, 1].grid(True, alpha=0.3)
    axes_state[0, 0].set_title("Position States")
    axes_state[0, 1].set_title("Velocity States")
    axes_state[-1, 0].set_xlabel("Time (s)")
    axes_state[-1, 1].set_xlabel("Time (s)")
    axes_state[0, 0].legend(loc="best")
    fig_state.suptitle(f"Orbit Propagation: Two-Body vs 8x8 Spherical Harmonics ({gravity_model} real coefficients)")
    fig_state.tight_layout()
    state_path = outdir / "state_two_body_vs_sh8x8.png"
    if plot_mode in ("save", "both"):
        fig_state.savefig(state_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_state)

    fig_eci = plt.figure(figsize=(9, 8))
    ax_eci = fig_eci.add_subplot(111, projection="3d")
    ax_eci.plot(x_two_body[:, 0], x_two_body[:, 1], x_two_body[:, 2], "--", linewidth=1.2, label="Two-body")
    ax_eci.plot(x_sh[:, 0], x_sh[:, 1], x_sh[:, 2], linewidth=1.4, label="8x8 SH")
    ax_eci.scatter([x_two_body[0, 0]], [x_two_body[0, 1]], [x_two_body[0, 2]], s=30, c="tab:green", label="Start")
    ax_eci.set_title("ECI Orbit Trajectories: Two-Body vs 8x8 Spherical Harmonics")
    ax_eci.set_xlabel("X (km)")
    ax_eci.set_ylabel("Y (km)")
    ax_eci.set_zlabel("Z (km)")
    ax_eci.grid(True, alpha=0.3)
    ax_eci.legend(loc="best")
    # Keep equal-ish scaling for spatial interpretation.
    r_stack = np.vstack((x_two_body[:, :3], x_sh[:, :3]))
    span = np.ptp(r_stack, axis=0)
    max_span = float(np.max(span))
    center = np.mean(r_stack, axis=0)
    half = 0.5 * max(max_span, 1e-6)
    ax_eci.set_xlim(center[0] - half, center[0] + half)
    ax_eci.set_ylim(center[1] - half, center[1] + half)
    ax_eci.set_zlim(center[2] - half, center[2] + half)
    try:
        ax_eci.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass
    fig_eci.tight_layout()
    eci_path = outdir / "eci_orbit_overlay_two_body_vs_sh8x8.png"
    if plot_mode in ("save", "both"):
        fig_eci.savefig(eci_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_eci)

    fig_err, axes_err = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    pos_comp_labels = ["dx (m)", "dy (m)", "dz (m)"]
    vel_comp_labels = ["dvx (mm/s)", "dvy (mm/s)", "dvz (mm/s)"]
    for i in range(3):
        axes_err[i, 0].plot(t, err[:, i] * 1e3, color="tab:blue")
        axes_err[i, 0].set_ylabel(pos_comp_labels[i])
        axes_err[i, 0].grid(True, alpha=0.3)
        axes_err[i, 1].plot(t, err[:, i + 3] * 1e6, color="tab:orange")
        axes_err[i, 1].set_ylabel(vel_comp_labels[i])
        axes_err[i, 1].grid(True, alpha=0.3)
    axes_err[3, 0].plot(t, pos_err_norm_m, color="tab:green")
    axes_err[3, 0].set_ylabel("|dr| (m)")
    axes_err[3, 0].grid(True, alpha=0.3)
    axes_err[3, 1].plot(t, vel_err_norm_mm_s, color="tab:red")
    axes_err[3, 1].set_ylabel("|dv| (mm/s)")
    axes_err[3, 1].grid(True, alpha=0.3)
    axes_err[0, 0].set_title("Position Error Components (8x8 SH - Two-Body)")
    axes_err[0, 1].set_title("Velocity Error Components (8x8 SH - Two-Body)")
    axes_err[-1, 0].set_xlabel("Time (s)")
    axes_err[-1, 1].set_xlabel("Time (s)")
    fig_err.tight_layout()
    err_path = outdir / "error_sh8x8_minus_two_body.png"
    if plot_mode in ("save", "both"):
        fig_err.savefig(err_path, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig_err)

    return {
        "plot_mode": plot_mode,
        "n_terms": str(len(terms)),
        "gravity_model": str(gravity_model),
        "coeff_path": str(coeff_path) if coeff_path else "(auto)",
        "inclination_deg": f"{float(inclination_deg):.3f}",
        "duration_s": f"{float(duration_s):.2f}",
        "max_pos_error_m": f"{float(np.max(pos_err_norm_m)):.6f}",
        "max_vel_error_mm_s": f"{float(np.max(vel_err_norm_mm_s)):.6f}",
        "eci_overlay_plot": str(eci_path) if plot_mode in ("save", "both") else "",
        "state_plot": str(state_path) if plot_mode in ("save", "both") else "",
        "error_plot": str(err_path) if plot_mode in ("save", "both") else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo propagation using 8x8 spherical harmonics (n<=8, m<=8).")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--dt", type=float, default=None, help="Integrator step size (s), overrides profile.")
    parser.add_argument("--altitude-km", type=float, default=500.0, help="Initial circular altitude (km).")
    parser.add_argument("--inclination-deg", type=float, default=45.0, help="Initial circular orbit inclination (deg).")
    parser.add_argument("--orbits", type=float, default=1.0, help="Propagation duration in orbit periods.")
    parser.add_argument("--gravity-model", type=str, default="EGM96", help="Real gravity model name (currently EGM96).")
    parser.add_argument("--coeff-path", type=str, default="", help="Optional local .gfc file path for real coefficients.")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Do not auto-download model file when coeff-path is not provided.",
    )
    parser.add_argument("--fd-step-km", type=float, default=1.0e-3, help="Finite-difference step for SH acceleration.")
    parser.add_argument(
        "--profile",
        choices=list(profile_choices()),
        default="ops",
        help="Fidelity profile: fast, ops, or high_fidelity.",
    )
    args = parser.parse_args()

    out = run_demo(
        plot_mode=args.plot_mode,
        dt_s=None if args.dt is None else float(args.dt),
        altitude_km=float(args.altitude_km),
        inclination_deg=float(args.inclination_deg),
        orbits=float(args.orbits),
        gravity_model=str(args.gravity_model),
        coeff_path=str(args.coeff_path),
        allow_download=not bool(args.no_download),
        fd_step_km=float(args.fd_step_km),
        profile=args.profile,
    )
    print("8x8 spherical harmonics demo outputs:")
    for k, v in out.items():
        if v:
            print(f"  {k}: {v}")
