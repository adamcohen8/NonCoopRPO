from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from presets import BASIC_SATELLITE, build_sim_object_from_presets
from sim.control.attitude import PoseCommandGenerator


def run_demo(
    dt_s: float = 1.0,
    lat_deg: float = 10.0,
    lon_deg: float = -45.0,
) -> dict[str, str]:
    sat = build_sim_object_from_presets(
        object_id="pose_demo_sat",
        dt_s=dt_s,
        satellite=BASIC_SATELLITE,
        enable_disturbances=False,
        enable_attitude_knowledge=False,
    )

    q_sun = PoseCommandGenerator.sun_track(
        truth=sat.truth,
        sun_dir_eci=np.array([1.0, 0.2, 0.1], dtype=float),
        panel_normal_body=np.array([0.0, 0.0, 1.0], dtype=float),
    )
    q_spot_geo = PoseCommandGenerator.spotlight_latlon(
        truth=sat.truth,
        latitude_deg=float(lat_deg),
        longitude_deg=float(lon_deg),
        altitude_km=0.0,
        boresight_body=np.array([1.0, 0.0, 0.0], dtype=float),
    )
    q_spot_ric = PoseCommandGenerator.spotlight_ric_direction(
        truth=sat.truth,
        ric_direction=np.array([0.0, 1.0, 0.0], dtype=float),  # In-track look direction
        boresight_body=np.array([1.0, 0.0, 0.0], dtype=float),
    )

    return {
        "sun_track_q_bn": str(q_sun.tolist()),
        "spotlight_latlon_q_bn": str(q_spot_geo.tolist()),
        "spotlight_ric_q_bn": str(q_spot_ric.tolist()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of mission pose commands that output target quaternions.")
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--lat-deg", type=float, default=10.0)
    parser.add_argument("--lon-deg", type=float, default=-45.0)
    args = parser.parse_args()

    out = run_demo(dt_s=float(args.dt), lat_deg=float(args.lat_deg), lon_deg=float(args.lon_deg))
    print("Pose command quaternion outputs:")
    for k, v in out.items():
        print(f"  {k}: {v}")
