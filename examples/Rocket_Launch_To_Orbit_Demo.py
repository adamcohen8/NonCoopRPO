from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.presets import BASIC_TWO_STAGE_STACK
from sim.rocket import (
    OpenLoopPitchProgramGuidance,
    RocketAeroConfig,
    RocketAscentSimulator,
    RocketSimConfig,
    RocketVehicleConfig,
    TVCSteeringGuidance,
)


def run_demo(plot_mode: str = "interactive") -> dict[str, str]:
    sim_cfg = RocketSimConfig(
        dt_s=0.5,
        max_time_s=2200.0,
        target_altitude_km=400.0,
        target_altitude_tolerance_km=30.0,
        target_eccentricity_max=0.05,
        insertion_hold_time_s=20.0,
        launch_lat_deg=28.5,
        launch_lon_deg=-80.6,
        launch_azimuth_deg=90.0,
        atmosphere_model="ussa1976",
        enable_drag=True,
        enable_srp=False,
        enable_j2=True,
        enable_j3=False,
        enable_j4=False,
        wind_enu_m_s=np.array([12.0, 0.0, 0.0]),
        tvc_time_constant_s=0.2,
        tvc_max_gimbal_deg=5.0,
        tvc_rate_limit_deg_s=8.0,
        tvc_pivot_offset_body_m=np.array([-6.0, 0.0, 0.0]),
        aero=RocketAeroConfig(
            enabled=True,
            reference_area_m2=10.0,
            reference_length_m=35.0,
            cp_offset_body_m=np.array([-2.5, 0.0, 0.0]),
            cd_base=0.18,
            cd_alpha2=0.08,
            cd_supersonic=0.28,
            transonic_peak_cd=0.25,
            transonic_mach=1.0,
            transonic_width=0.20,
            cl_alpha_per_rad=0.12,
            cy_beta_per_rad=0.12,
            cm_alpha_per_rad=-0.02,
            cn_beta_per_rad=-0.02,
        ),
    )
    vehicle_cfg = RocketVehicleConfig(
        stack=BASIC_TWO_STAGE_STACK,
        payload_mass_kg=12000.0,
        thrust_axis_body=np.array([1.0, 0.0, 0.0]),
    )
    guidance = TVCSteeringGuidance(
        base_guidance=OpenLoopPitchProgramGuidance(
            vertical_hold_s=8.0,
            pitch_start_s=8.0,
            pitch_end_s=190.0,
            pitch_final_deg=75.0,
            max_throttle=1.0,
        ),
        pass_through_attitude=True,
    )

    sim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=guidance)
    out = sim.run()

    t = out.time_s
    alt = out.altitude_km
    speed = np.linalg.norm(out.velocity_eci_km_s, axis=1)
    mass = out.mass_kg
    stage = out.active_stage_index
    ecc = out.eccentricity
    sma = out.sma_km
    q_dyn = out.dynamic_pressure_pa
    mach = out.mach
    lat = out.latitude_deg
    lon = out.longitude_deg
    wind_mag = np.linalg.norm(out.wind_body_m_s, axis=1)
    gimbal_deg = out.tvc_gimbal_deg
    alpha_deg = out.alpha_deg
    beta_deg = out.beta_deg
    cd_hist = out.cd
    aero_force_n = out.aero_force_n
    aero_moment_nm = out.aero_moment_nm

    outdir = REPO_ROOT / "outputs" / "rocket_launch_demo"
    if plot_mode in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    ax[0, 0].plot(t, alt)
    ax[0, 0].set_ylabel("km")
    ax[0, 0].set_title("Altitude")
    ax[0, 0].grid(True, alpha=0.3)

    ax[0, 1].plot(t, speed)
    ax[0, 1].set_ylabel("km/s")
    ax[0, 1].set_title("Speed")
    ax[0, 1].grid(True, alpha=0.3)

    ax[1, 0].plot(t, mass)
    ax[1, 0].set_ylabel("kg")
    ax[1, 0].set_title("Mass")
    ax[1, 0].grid(True, alpha=0.3)

    ax[1, 1].plot(t, stage)
    ax[1, 1].set_ylabel("index")
    ax[1, 1].set_title("Active Stage Index")
    ax[1, 1].grid(True, alpha=0.3)

    ax[2, 0].plot(t, ecc)
    ax[2, 0].set_ylabel("-")
    ax[2, 0].set_title("Eccentricity")
    ax[2, 0].grid(True, alpha=0.3)
    ax[2, 0].set_xlabel("Time (s)")

    ax[2, 1].plot(t, sma)
    ax[2, 1].set_ylabel("km")
    ax[2, 1].set_title("Semi-major Axis")
    ax[2, 1].grid(True, alpha=0.3)
    ax[2, 1].set_xlabel("Time (s)")

    fig.suptitle(
        f"Rocket Ascent Demo (TVC-Assisted Pitch Program) | Inserted={out.inserted} | t_insert={out.insertion_time_s if out.insertion_time_s is not None else 'n/a'}"
    )
    fig.tight_layout()
    p = outdir / "rocket_launch_profiles.png"
    if plot_mode in ("save", "both"):
        fig.savefig(p, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    fig2, ax2 = plt.subplots(4, 2, figsize=(12, 11), sharex=True)
    ax2[0, 0].plot(t, q_dyn)
    ax2[0, 0].set_ylabel("Pa")
    ax2[0, 0].set_title("Dynamic Pressure")
    ax2[0, 0].grid(True, alpha=0.3)

    ax2[0, 1].plot(t, mach)
    ax2[0, 1].set_ylabel("-")
    ax2[0, 1].set_title("Mach")
    ax2[0, 1].grid(True, alpha=0.3)

    ax2[1, 0].plot(t, alpha_deg)
    ax2[1, 0].set_ylabel("deg")
    ax2[1, 0].set_title("Alpha")
    ax2[1, 0].grid(True, alpha=0.3)

    ax2[1, 1].plot(t, beta_deg)
    ax2[1, 1].set_ylabel("deg")
    ax2[1, 1].set_title("Beta")
    ax2[1, 1].grid(True, alpha=0.3)

    ax2[2, 0].plot(t, cd_hist)
    ax2[2, 0].set_ylabel("-")
    ax2[2, 0].set_title("Cd")
    ax2[2, 0].grid(True, alpha=0.3)

    ax2[2, 1].plot(t, aero_force_n)
    ax2[2, 1].set_ylabel("N")
    ax2[2, 1].set_title("|Aerodynamic Force|")
    ax2[2, 1].grid(True, alpha=0.3)

    ax2[3, 0].plot(t, aero_moment_nm)
    ax2[3, 0].set_ylabel("N m")
    ax2[3, 0].set_title("|Aerodynamic Moment|")
    ax2[3, 0].grid(True, alpha=0.3)
    ax2[3, 0].set_xlabel("Time (s)")

    ax2[3, 1].axis("off")
    fig2.suptitle("Rocket Aerodynamic Diagnostics")
    fig2.tight_layout()
    p2 = outdir / "rocket_aero_profiles.png"
    if plot_mode in ("save", "both"):
        fig2.savefig(p2, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig2)

    fig3, ax3 = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    ax3[0, 0].plot(t, lat)
    ax3[0, 0].set_ylabel("deg")
    ax3[0, 0].set_title("Latitude")
    ax3[0, 0].grid(True, alpha=0.3)

    ax3[0, 1].plot(t, lon)
    ax3[0, 1].set_ylabel("deg")
    ax3[0, 1].set_title("Longitude")
    ax3[0, 1].grid(True, alpha=0.3)

    ax3[1, 0].plot(t, wind_mag)
    ax3[1, 0].set_ylabel("m/s")
    ax3[1, 0].set_title("|Wind Body|")
    ax3[1, 0].grid(True, alpha=0.3)

    ax3[1, 1].plot(t, gimbal_deg)
    ax3[1, 1].set_ylabel("deg")
    ax3[1, 1].set_title("TVC Gimbal")
    ax3[1, 1].grid(True, alpha=0.3)

    ax3[2, 0].plot(t, out.throttle_cmd)
    ax3[2, 0].set_ylabel("Throttle")
    ax3[2, 0].set_title("Throttle Command")
    ax3[2, 0].grid(True, alpha=0.3)
    ax3[2, 0].set_xlabel("Time (s)")

    ax3[2, 1].plot(t, out.thrust_n)
    ax3[2, 1].set_ylabel("N")
    ax3[2, 1].set_title("Engine Thrust")
    ax3[2, 1].grid(True, alpha=0.3)
    ax3[2, 1].set_xlabel("Time (s)")

    fig3.suptitle("Rocket Guidance and Environment Diagnostics")
    fig3.tight_layout()
    p3 = outdir / "rocket_guidance_env_profiles.png"
    if plot_mode in ("save", "both"):
        fig3.savefig(p3, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig3)

    return {
        "inserted": str(out.inserted),
        "insertion_time_s": "" if out.insertion_time_s is None else f"{out.insertion_time_s:.2f}",
        "final_altitude_km": f"{alt[-1]:.3f}",
        "final_eccentricity": f"{ecc[-1]:.5f}",
        "plot": str(p) if plot_mode in ("save", "both") else "",
        "aero_plot": str(p2) if plot_mode in ("save", "both") else "",
        "guidance_env_plot": str(p3) if plot_mode in ("save", "both") else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rocket launch-to-orbit demo (dedicated rocket sim engine).")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    args = parser.parse_args()
    result = run_demo(plot_mode=args.plot_mode)
    print("Rocket ascent demo outputs:")
    for k, v in result.items():
        if v:
            print(f"  {k}: {v}")
