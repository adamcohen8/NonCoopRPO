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
    MaxQThrottleLimiterGuidance,
    OpenLoopPitchProgramGuidance,
    RocketAeroConfig,
    RocketAscentSimulator,
    RocketSimConfig,
    RocketVehicleConfig,
)


def run_demo(plot_mode: str = "interactive", max_q_pa: float = 30_000.0) -> dict[str, str]:
    sim_cfg = RocketSimConfig(
        dt_s=0.5,
        max_time_s=900.0,
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
    base_guidance = OpenLoopPitchProgramGuidance(
        vertical_hold_s=8.0,
        pitch_start_s=8.0,
        pitch_end_s=190.0,
        pitch_final_deg=75.0,
        max_throttle=1.0,
    )
    limited_guidance = MaxQThrottleLimiterGuidance(
        base_guidance=base_guidance,
        max_q_pa=max_q_pa,
        min_throttle=0.0,
    )

    out_nom = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=base_guidance).run()
    out_lim = RocketAscentSimulator(sim_cfg=sim_cfg, vehicle_cfg=vehicle_cfg, guidance=limited_guidance).run()

    outdir = REPO_ROOT / "outputs" / "rocket_maxq_compare_demo"
    if plot_mode in ("save", "both"):
        outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    ax[0, 0].plot(out_nom.time_s, out_nom.dynamic_pressure_pa, label="Nominal")
    ax[0, 0].plot(out_lim.time_s, out_lim.dynamic_pressure_pa, label="Max-Q Limited")
    ax[0, 0].axhline(max_q_pa, color="k", linestyle="--", linewidth=1.0, label="Max-Q cap")
    ax[0, 0].set_title("Dynamic Pressure")
    ax[0, 0].set_ylabel("Pa")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].legend(loc="best")

    ax[0, 1].plot(out_nom.time_s, out_nom.throttle_cmd, label="Nominal")
    ax[0, 1].plot(out_lim.time_s, out_lim.throttle_cmd, label="Max-Q Limited")
    ax[0, 1].set_title("Throttle Command")
    ax[0, 1].set_ylabel("Throttle")
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].legend(loc="best")

    ax[1, 0].plot(out_nom.time_s, out_nom.altitude_km, label="Nominal")
    ax[1, 0].plot(out_lim.time_s, out_lim.altitude_km, label="Max-Q Limited")
    ax[1, 0].set_title("Altitude")
    ax[1, 0].set_ylabel("km")
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 0].grid(True, alpha=0.3)
    ax[1, 0].legend(loc="best")

    ax[1, 1].plot(out_nom.time_s, np.linalg.norm(out_nom.velocity_eci_km_s, axis=1), label="Nominal")
    ax[1, 1].plot(out_lim.time_s, np.linalg.norm(out_lim.velocity_eci_km_s, axis=1), label="Max-Q Limited")
    ax[1, 1].set_title("Speed")
    ax[1, 1].set_ylabel("km/s")
    ax[1, 1].set_xlabel("Time (s)")
    ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].legend(loc="best")
    fig.suptitle("Rocket Ascent: Nominal vs Max-Q Throttle Limiter")
    fig.tight_layout()

    p = outdir / "rocket_maxq_compare.png"
    if plot_mode in ("save", "both"):
        fig.savefig(p, dpi=160)
    if plot_mode in ("interactive", "both"):
        plt.show()
    plt.close(fig)

    qpk_nom = float(np.max(out_nom.dynamic_pressure_pa))
    qpk_lim = float(np.max(out_lim.dynamic_pressure_pa))
    return {
        "max_q_target_pa": f"{max_q_pa:.1f}",
        "q_peak_nominal_pa": f"{qpk_nom:.1f}",
        "q_peak_limited_pa": f"{qpk_lim:.1f}",
        "plot": str(p) if plot_mode in ("save", "both") else "",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare nominal rocket ascent against max-Q-throttle-limited ascent.")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--max-q-pa", type=float, default=30_000.0)
    args = parser.parse_args()
    out = run_demo(plot_mode=args.plot_mode, max_q_pa=float(args.max_q_pa))
    print("Rocket max-Q compare outputs:")
    for k, v in out.items():
        if v:
            print(f"  {k}: {v}")
