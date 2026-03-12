from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.scenarios import run_free_tumble_one_orbit
from sim.utils import (
    animate_rectangular_prism_attitude,
    plot_body_rates,
    plot_control_commands,
    plot_quaternion_components,
    plot_trajectory_frame,
)


def run_demo(plot_mode: str = "interactive", include_animation: bool = False) -> dict[str, str]:
    out = run_free_tumble_one_orbit(output_dir=str(REPO_ROOT / "outputs" / "plotting_capabilities_demo"), plot_mode="save")
    # Reuse the produced log for plotting capability demos.
    import json

    log_path = Path(out["log_json"])
    with log_path.open("r", encoding="utf-8") as f:
        d = json.load(f)
    t = np.array(d["t_s"], dtype=float)
    truth = np.array(d["truth_by_object"]["sat_01"], dtype=float)
    outdir = REPO_ROOT / "outputs" / "plotting_capabilities_demo"
    outdir.mkdir(parents=True, exist_ok=True)

    plot_quaternion_components(
        t,
        truth,
        frame="eci",
        layout="single",
        mode=plot_mode,
        out_path=str(outdir / "quat_eci_single.png"),
    )
    plot_quaternion_components(
        t,
        truth,
        frame="ric",
        layout="subplots",
        mode=plot_mode,
        out_path=str(outdir / "quat_ric_subplots.png"),
    )
    plot_body_rates(
        t,
        truth,
        frame="eci",
        layout="subplots",
        mode=plot_mode,
        out_path=str(outdir / "rates_eci_subplots.png"),
    )
    plot_body_rates(
        t,
        truth,
        frame="ric",
        layout="single",
        mode=plot_mode,
        out_path=str(outdir / "rates_ric_single.png"),
    )
    plot_trajectory_frame(
        t,
        truth,
        frame="eci",
        mode=plot_mode,
        out_path=str(outdir / "traj_eci.png"),
    )
    plot_trajectory_frame(
        t,
        truth,
        frame="ecef",
        mode=plot_mode,
        out_path=str(outdir / "traj_ecef.png"),
    )
    plot_control_commands(
        t,
        np.array(d["applied_thrust_by_object"]["sat_01"], dtype=float),
        layout="subplots",
        input_labels=["ax", "ay", "az"],
        title="Applied Thrust Commands",
        y_label="km/s^2",
        mode=plot_mode,
        out_path=str(outdir / "thrust_commands.png"),
    )
    if include_animation:
        animate_rectangular_prism_attitude(
            t,
            truth,
            lx_m=2.0,
            ly_m=1.0,
            lz_m=0.8,
            frame="eci",
            mode=plot_mode,
            out_path=str(outdir / "rect_attitude_eci.mp4"),
        )
    return {
        "output_dir": str(outdir),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo for expanded plotting capabilities.")
    parser.add_argument("--plot-mode", choices=["interactive", "save", "both"], default="interactive")
    parser.add_argument("--include-animation", action="store_true")
    args = parser.parse_args()
    res = run_demo(plot_mode=args.plot_mode, include_animation=bool(args.include_animation))
    print("Plotting capability demo outputs:")
    for k, v in res.items():
        print(f"  {k}: {v}")
