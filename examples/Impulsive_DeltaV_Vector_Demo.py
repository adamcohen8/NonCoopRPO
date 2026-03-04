from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from presets import build_sim_object_from_presets
from sim.control.orbit.impulsive import AttitudeAgnosticImpulsiveManeuverer, DeltaVManeuver


if __name__ == "__main__":
    sat = build_sim_object_from_presets(object_id="sat_impulse_dv_demo", dt_s=2.0)
    truth = sat.truth

    available_dv_km_s = 0.05
    dv_cmd = np.array([0.03, -0.02, 0.01])

    maneuverer = AttitudeAgnosticImpulsiveManeuverer()
    new_truth, result = maneuverer.execute_delta_v(
        truth=truth,
        maneuver=DeltaVManeuver(delta_v_eci_km_s=dv_cmd),
        available_delta_v_km_s=available_dv_km_s,
    )

    print("executed:", result.executed)
    print("required_dv_km_s:", result.required_delta_v_km_s)
    print("remaining_dv_km_s:", result.remaining_delta_v_km_s)
    print("old_velocity_km_s:", truth.velocity_eci_km_s)
    print("dv_cmd_km_s:", dv_cmd)
    print("new_velocity_km_s:", new_truth.velocity_eci_km_s)
